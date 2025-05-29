import chalk from "chalk";

export class Logger {
    constructor(verbose = false) {
        this.verbose = verbose;
    }

    info(message, ...args) {
        console.log(chalk.blue("ℹ"), message, ...args);
    }

    success(message, ...args) {
        console.log(chalk.green("✓"), message, ...args);
    }

    warn(message, ...args) {
        console.log(chalk.yellow("⚠"), message, ...args);
    }

    error(message, ...args) {
        console.log(chalk.red("✗"), message, ...args);
    }

    debug(message, ...args) {
        if (this.verbose) {
            console.log(chalk.gray("🔍"), message, ...args);
        }
    }
}
