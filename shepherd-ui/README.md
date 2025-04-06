# Angular 19 Project - Documentation

## Project Overview

This Angular 19 project is built with a modern development workflow that includes:

- **Angular Material 3 (M3)** for UI components and a custom Material theme.
- **ESLint** for ensuring TypeScript code quality.
- **Stylelint** for SCSS linting.
- **Prettier** for consistent code formatting.
- **Husky** and **lint-staged** to run pre-commit checks.
- **Commitlint** to enforce Conventional Commit messages.

---

## Prerequisites

Ensure you have the following installed:

- **Node.js** (v22 or later recommended)
- **npm** (v10 or later)
- **Angular CLI** (v19.1.1 or later)

---

## Installation

Clone the repository and install dependencies:

```bash
git clone https://github.com/your-org/your-repo.git
cd your-repo
npm install
```

---

## Development Server

Start the development server with:

```bash
ng serve
```

Then open [http://localhost:4200](http://localhost:4200) in your browser.

---

## Code Quality and Formatting

### ESLint for TypeScript

ESLint checks the quality of your TypeScript code. To run ESLint, execute:

```bash
npm run lint
```

### Stylelint for SCSS

Stylelint checks your SCSS code against best practices and coding standards. The configuration is defined in **.stylelintrc.json** at the project root. To auto-fix issues, run:

```bash
npx stylelint --fix "src/**/*.scss"
```

### Prettier for Code Formatting

Prettier ensures a consistent code format. The rules are defined in the **.prettierrc** file at the project root. To format all SCSS files, run:

```bash
npx prettier --write "src/**/*.scss"
```

### Combined Commands

You can run both tools using custom scripts defined in the **package.json**:

```json
"scripts": {
  "format:scss": "npx prettier --write \"src/**/*.scss\"",
  "lint:scss": "npx stylelint --fix \"src/**/*.scss\"",
  "format:all": "npm run format:scss && npm run lint:scss"
}
```

This allows you to format and lint your SCSS files manually by running:

```bash
npm run format:all
```

---

## Angular Material 3 (M3) Setup

This project uses Angular Material 3 for its UI components along with a custom Material theme (configured in the project’s styles). When building your components, import the necessary Angular Material modules and apply the theme as needed. This setup ensures a modern, responsive, and accessible user interface.

---

## Git Hooks and Commit Message Enforcement

### Husky and lint-staged

Husky is configured to run pre-commit hooks that trigger linting and formatting on only the staged files. The **lint-staged** configuration in the **package.json** is:

```json
"lint-staged": {
  "*.ts": [
    "eslint --fix"
  ],
  "*.tsx": [
    "eslint --fix"
  ],
  "*.scss": [
    "npx prettier --write",
    "npx stylelint --fix --"
  ]
}
```

*Note:* The `--` at the end of the Stylelint command helps separate options from file paths, which is particularly helpful on Windows.

### Commitlint

Commitlint enforces Conventional Commit message guidelines. The configuration is in **commitlint.config.js**. Commit messages should follow the format (e.g., `feat: add new feature`). If your message doesn’t meet these guidelines, Commitlint will block the commit and display an error.

---

## Building and Testing

### Build

To build the project for production, run:

```bash
ng build
```

The build artifacts will be output to the `dist/` directory.

### Testing

Run unit tests using Karma and Jasmine:

```bash
npm run test
```

Test files are located alongside the components (with the `.spec.ts` extension).

---

## Additional Notes

- **Git Hooks:** Husky automatically runs pre-commit and commit-msg hooks. Do not modify the files in the **.husky** directory unless necessary.
- **Commit Messages:** Follow the Conventional Commits format to ensure your commits pass validation.
- **Workflow Integration:** ESLint, Stylelint, and Prettier are integrated via lint-staged, ensuring that only staged files are checked and formatted before commits.
- **Angular Material 3:** The project leverages Angular Material 3 for UI components. Make sure to import and use the Material modules as needed in your components.

---

## License

This project is licensed under the **MIT License**.

---