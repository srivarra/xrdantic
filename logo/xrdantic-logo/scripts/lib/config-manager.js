import { readFileSync } from "fs";
import { join } from "path";

export class ConfigManager {
    constructor(scriptsDir, logger) {
        this.scriptsDir = scriptsDir;
        this.logger = logger;
        this.templatesDir = join(scriptsDir, "templates");
    }

    async loadConfig(templateName) {
        try {
            const configPath = join(
                this.templatesDir,
                templateName,
                "config.json",
            );
            const configContent = readFileSync(configPath, "utf8");
            const config = JSON.parse(configContent);

            this.logger.debug(`Loaded config for template: ${templateName}`);
            return {
                template: templateName,
                ...config,
            };
        } catch (error) {
            throw new Error(
                `Failed to load config for template '${templateName}': ${error.message}`,
            );
        }
    }

    getTemplatePath(templateName) {
        return join(this.templatesDir, templateName);
    }
}
