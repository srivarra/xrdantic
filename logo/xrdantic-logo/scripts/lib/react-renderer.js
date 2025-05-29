import React from "react";
import { renderToStaticMarkup } from "react-dom/server";

export class ReactRenderer {
    constructor(logger) {
        this.logger = logger;
    }

    async renderToSvg(TemplateComponent, config) {
        try {
            this.logger.debug("Rendering React component to SVG...");

            // Create React element with config as props
            const element = React.createElement(TemplateComponent, config);

            // Render to static markup
            const markup = renderToStaticMarkup(element);

            this.logger.debug("React component rendered successfully");
            return markup;
        } catch (error) {
            throw new Error(
                `Failed to render React component: ${error.message}`,
            );
        }
    }

    async renderToHtml(TemplateComponent, config) {
        try {
            const svgMarkup = await this.renderToSvg(TemplateComponent, config);

            return `
        <!DOCTYPE html>
        <html>
          <head>
            <meta charset="utf-8">
            <title>SVG Preview</title>
            <style>
              body { margin: 0; padding: 20px; background: #f0f0f0; }
              svg { background: white; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }
            </style>
          </head>
          <body>
            ${svgMarkup}
          </body>
        </html>
      `;
        } catch (error) {
            throw new Error(`Failed to render HTML preview: ${error.message}`);
        }
    }
}
