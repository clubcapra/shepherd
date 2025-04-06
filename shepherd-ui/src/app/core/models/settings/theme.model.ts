export enum ThemeType {
  DARK = 'dark',
  LIGHT = 'light'
}
export type ThemeConstantType<T = string> = Record<ThemeType, T>
export interface ThemeSummaryModel {
  value: string,
  icon: string
  i18n: string
}
