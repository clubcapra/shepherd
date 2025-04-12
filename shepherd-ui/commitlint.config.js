module.exports = {
    extends: ['@commitlint/config-conventional'],
    rules: {
        'header-min-length': [2, 'always', 10],
        'header-max-length': [2, 'always', 72],
        'type-empty': [0],
        'subject-empty': [0]
    }
};
