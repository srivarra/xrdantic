import { program } from "commander";
import chalk from "chalk";
import ora from "ora";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync, mkdirSync } from "fs";
import chokidar from "chokidar";

import { SvgLoader } from "./lib/svg-loader.js";
import { ReactRenderer } from "./lib/react-renderer.js";
import { PngConverter } from "./lib/png-converter.js";
import { ConfigManager } from "./lib/config-manager.js";
import { Logger } from "./lib/logger.js";

const __filename = fileURLToPath(import.meta.url);
const __dirname = dirname(__filename);

// Configure CLI
program
    .name("svg-to-png-converter")
    .description("Convert SVG to PNG with React rendering and revision control")
    .version("1.0.0")
    .option("-r, --revision <version>", "Revision/version number", "0.0.1")
    .option("-t, --template <name>", "Template name to use", "xrdantic")
    .option("-o, --output <path>", "Output file path", "dist/banner.png")
    .option("-w, --width <number>", "Output width in pixels", "1280")
    .option("-h, --height <number>", "Output height in pixels", "640")
    .option("--watch", "Watch for file changes and rebuild")
    .option("--quality <number>", "PNG quality (1-9)", "9")
    .option("--verbose", "Verbose logging")
    .option("--dry-run", "Show what would be done without executing");

program.parse();

const options = program.opts();

// Initialize services
const logger = new Logger(options.verbose);
const configManager = new ConfigManager(__dirname, logger);
const svgLoader = new SvgLoader(logger);
const reactRenderer = new ReactRenderer(logger);
const pngConverter = new PngConverter(logger);

async function buildBanner() {
    const spinner = ora("Building banner...").start();

    try {
        // Load configuration
        const config = await configManager.loadConfig(options.template);

        // Merge CLI options with config
        const buildConfig = {
            ...config,
            revision: options.revision,
            output: options.output,
            width: Number.parseInt(options.width),
            height: Number.parseInt(options.height),
            quality: Number.parseInt(options.quality),
        };

        logger.info("Build configuration:", buildConfig);

        if (options.dryRun) {
            spinner.succeed("Dry run completed - no files were generated");
            return;
        }

        // Ensure output directory exists
        const outputDir = dirname(buildConfig.output);
        if (!existsSync(outputDir)) {
            mkdirSync(outputDir, { recursive: true });
        }

        // Step 1: Load SVG template
        spinner.text = "Loading SVG template...";
        const svgTemplate = await svgLoader.loadTemplate(buildConfig.template);

        // Step 2: Render React component to SVG
        spinner.text = "Rendering React component...";
        const renderedSvg = await reactRenderer.renderToSvg(
            svgTemplate,
            buildConfig,
        );

        // Step 3: Convert to PNG
        spinner.text = "Converting to PNG...";
        const pngBuffer = await pngConverter.convert(renderedSvg, buildConfig);

        // Step 4: Save output
        spinner.text = "Saving output...";
        await pngConverter.save(pngBuffer, buildConfig.output);

        const stats = await pngConverter.getStats(buildConfig.output);

        spinner.succeed(chalk.green("Banner generated successfully!"));

        logger.success("Output details:");
        logger.info(`  File: ${buildConfig.output}`);
        logger.info(`  Size: ${stats.width}x${stats.height}px`);
        logger.info(`  File size: ${(stats.size / 1024).toFixed(2)} KB`);
        logger.info(`  Revision: ${buildConfig.revision}`);
    } catch (error) {
        spinner.fail(chalk.red("Build failed"));
        logger.error("Error:", error.message);
        if (options.verbose) {
            console.error(error.stack);
        }
        process.exit(1);
    }
}

async function watchMode() {
    logger.info("Starting watch mode...");

    const watchPaths = [
        join(__dirname, "templates/**/*.js"),
        join(__dirname, "templates/**/*.json"),
        join(__dirname, "components/**/*.js"),
    ];

    const watcher = chokidar.watch(watchPaths, {
        ignored: /node_modules/,
        persistent: true,
    });

    watcher.on("change", async (path) => {
        logger.info(`File changed: ${path}`);
        await buildBanner();
    });

    watcher.on("ready", () => {
        logger.success("Watching for changes...");
        buildBanner();
    });

    // Handle graceful shutdown
    process.on("SIGINT", () => {
        logger.info("Shutting down watcher...");
        watcher.close();
        process.exit(0);
    });
}

// Main execution
if (options.watch) {
    watchMode();
} else {
    buildBanner();
}
