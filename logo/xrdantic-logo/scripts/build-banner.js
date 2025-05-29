import { program } from "commander";
import chalk from "chalk";
import ora from "ora";
import { fileURLToPath } from "url";
import { dirname, join } from "path";
import { existsSync, mkdirSync, writeFileSync } from "fs";
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
    .option(
        "-o, --output <path>",
        "Output file path (extension determines PNG or SVG if format is not specified)",
        "dist/banner.png",
    )
    .option("-w, --width <number>", "Output width in pixels", "1280")
    .option("-h, --height <number>", "Output height in pixels", "640")
    .option("--watch", "Watch for file changes and rebuild")
    .option("--quality <number>", "PNG quality (1-9)", "9")
    .option(
        "--output-format <format>",
        "Output format (svg, png, both)",
        "both",
    )
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
            output: options.output, // Base output path, extension might be ignored depending on format
            width: Number.parseInt(options.width),
            height: Number.parseInt(options.height),
            quality: Number.parseInt(options.quality),
            outputFormat: options.outputFormat.toLowerCase(),
        };

        logger.info("Build configuration:", buildConfig);

        if (options.dryRun) {
            spinner.succeed("Dry run completed - no files were generated");
            return;
        }

        // Ensure output directory exists
        const baseOutputName = buildConfig.output.includes(".")
            ? buildConfig.output.substring(
                  0,
                  buildConfig.output.lastIndexOf("."),
              )
            : buildConfig.output;
        const outputDir = dirname(baseOutputName);
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

        let svgOutputPath = null;
        let pngOutputPath = null;
        let pngStats = null;

        if (
            buildConfig.outputFormat === "svg" ||
            buildConfig.outputFormat === "both"
        ) {
            spinner.text = "Saving SVG output...";
            svgOutputPath = `${baseOutputName}.svg`;
            try {
                writeFileSync(svgOutputPath, renderedSvg, "utf8");
                logger.info(`SVG saved to: ${svgOutputPath}`);
            } catch (error) {
                spinner.warn(chalk.yellow("SVG saving failed."));
                logger.warn("Could not save SVG file:", error.message);
            }
        }

        if (
            buildConfig.outputFormat === "png" ||
            buildConfig.outputFormat === "both"
        ) {
            // Step 3: Convert to PNG
            spinner.text = "Converting to PNG...";
            const pngBuffer = await pngConverter.convert(
                renderedSvg,
                buildConfig,
            );

            // Step 4: Save output
            spinner.text = "Saving PNG output...";
            pngOutputPath = `${baseOutputName}.png`;
            await pngConverter.save(pngBuffer, pngOutputPath);
            pngStats = await pngConverter.getStats(pngOutputPath);
        }

        if (svgOutputPath || pngOutputPath) {
            spinner.succeed(chalk.green("Banner generated successfully!"));
            logger.success("Output details:");
            if (pngOutputPath && pngStats) {
                logger.info(`  PNG File: ${pngOutputPath}`);
                logger.info(`  Size: ${pngStats.width}x${pngStats.height}px`);
                logger.info(
                    `  File size (PNG): ${(pngStats.size / 1024).toFixed(2)} KB`,
                );
            }
            if (svgOutputPath) {
                logger.info(`  SVG File: ${svgOutputPath}`);
            }
            logger.info(`  Revision: ${buildConfig.revision}`);
        } else {
            spinner.warn(
                chalk.yellow(
                    "No output generated based on the format specified.",
                ),
            );
        }
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
        join(__dirname, "templates/**/*.jsx"), // Assuming templates are .jsx
        join(__dirname, "templates/**/*.json"),
        join(__dirname, "components/**/*.jsx"), // Assuming components are .jsx
    ];

    const watcher = chokidar.watch(watchPaths, {
        ignored: /node_modules/,
        persistent: true,
    });

    watcher.on("change", async (path) => {
        logger.info(`File changed: ${path}`);
        await buildBanner(); // Rebuild with current options
    });

    watcher.on("ready", () => {
        logger.success("Watching for changes...");
        buildBanner(); // Initial build with current options
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
