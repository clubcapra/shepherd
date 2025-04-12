
export enum LanguageType  {
  EN = 'en',
  FR = 'fr'
}
export type LanguageConstantType<T = string> = Record<LanguageType, T>
export interface LanguageSummaryModel {
  value: LanguageType,
  icon: string,
  i18n: string,
  label: string
}
