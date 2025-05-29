import sharp from "sharp";
import { writeFileSync, statSync } from "fs";
import puppeteer from "puppeteer";

export class PngConverter {
    constructor(logger) {
        this.logger = logger;
    }

    async convert(svgContent, config) {
        try {
            const { width, height, quality } = config;

            this.logger.debug(`Converting SVG to PNG (${width}x${height})`);

            // Method 1: Direct Sharp conversion (faster)
            if (this.isSimpleSvg(svgContent)) {
                return await this.convertWithSharp(svgContent, config);
            }

            // Method 2: Puppeteer conversion (for complex SVGs with web fonts, etc.)
            return await this.convertWithPuppeteer(svgContent, config);
        } catch (error) {
            throw new Error(`PNG conversion failed: ${error.message}`);
        }
    }

    async convertWithSharp(svgContent, config) {
        const { width, height, quality } = config;

        return await sharp(Buffer.from(svgContent))
            .resize(width, height)
            .png({
                compressionLevel: quality,
                adaptiveFiltering: true,
                force: true,
            })
            .toBuffer();
    }

    async convertWithPuppeteer(svgContent, config) {
        const { width, height } = config;

        this.logger.debug("Using Puppeteer for complex SVG rendering...");

        const browser = await puppeteer.launch({
            headless: "new",
            args: ["--no-sandbox", "--disable-setuid-sandbox"],
            // No executablePath, so Puppeteer uses its bundled browser
        });

        try {
            const page = await browser.newPage();
            await page.setViewport({ width, height, deviceScaleFactor: 1 });

            const html = `
        <!DOCTYPE html>
        <html>
          <head>
            <style>
              body { margin: 0; padding: 0; }
              svg { display: block; }
            </style>
          </head>
          <body>${svgContent}</body>
        </html>
      `;

            await page.setContent(html);
            await page.waitForSelector("svg");

            const pngBuffer = await page.screenshot({
                type: "png",
                omitBackground: true,
                clip: { x: 0, y: 0, width, height },
            });

            return pngBuffer;
        } finally {
            await browser.close();
        }
    }

    isSimpleSvg(svgContent) {
        // Check if SVG uses complex features that might need browser rendering
        const complexFeatures = [
            "foreignObject",
            "@import",
            "font-family.*Geist",
            "filter:",
            "mask:",
            "clip-path:",
        ];

        return !complexFeatures.some((feature) =>
            new RegExp(feature, "i").test(svgContent),
        );
    }

    async save(buffer, outputPath) {
        try {
            writeFileSync(outputPath, buffer);
            this.logger.debug(`PNG saved to: ${outputPath}`);
        } catch (error) {
            throw new Error(`Failed to save PNG: ${error.message}`);
        }
    }

    async getStats(filePath) {
        try {
            const stats = statSync(filePath);
            const metadata = await sharp(filePath).metadata();

            return {
                size: stats.size,
                width: metadata.width,
                height: metadata.height,
                format: metadata.format,
            };
        } catch (error) {
            throw new Error(`Failed to get file stats: ${error.message}`);
        }
    }
}
