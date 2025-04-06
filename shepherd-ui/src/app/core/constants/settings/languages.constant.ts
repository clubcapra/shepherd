import {LanguageConstantType, LanguageSummaryModel, LanguageType} from '@core/models/settings/language.model';

export const LanguagesConstant = Object.values(LanguageType);

export const LanguagesFlagConstant: LanguageConstantType = {
  en: 'gb',
  fr: 'fr'
};

export const LanguagesLabelConstant: LanguageConstantType = {
  en: 'English',
  fr: 'Français'
};

export const LanguagesI18NConstant: LanguageConstantType = {
  en: 'language.en'.toUpperCase(),
  fr: 'language.fr'.toUpperCase()
};

export const LanguagesSummaryConstant: LanguageSummaryModel[] = Object.values(LanguageType).map(t => ({
  value: t,
  icon: LanguagesFlagConstant[t],
  i18n: LanguagesI18NConstant[t],
  label: LanguagesLabelConstant[t]
}));
