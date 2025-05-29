import { join } from "path";

export class SvgLoader {
    constructor(logger) {
        this.logger = logger;
    }

    async loadTemplate(templateName) {
        try {
            const templatePath = join(
                process.cwd(),
                "scripts",
                "templates",
                templateName,
                "template.jsx",
            );

            // Dynamic import for ES modules
            const templateModule = await import(templatePath);

            this.logger.debug(`Loaded SVG template: ${templateName}`);
            return templateModule.default || templateModule;
        } catch (error) {
            throw new Error(
                `Failed to load SVG template '${templateName}': ${error.message}`,
            );
        }
    }

    validateSvg(svgContent) {
        if (!svgContent || typeof svgContent !== "string") {
            throw new Error("Invalid SVG content");
        }

        if (!svgContent.includes("<svg")) {
            throw new Error("Content does not appear to be valid SVG");
        }

        return true;
    }
}
