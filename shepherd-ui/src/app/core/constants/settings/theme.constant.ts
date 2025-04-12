import {ThemeConstantType, ThemeSummaryModel, ThemeType} from '@core/models/settings/theme.model';

export const ThemeConstant = Object.values(ThemeType)
export const ThemeI18NConstant: ThemeConstantType = {
  light: 'theme.light'.toUpperCase(),
  dark: 'theme.dark'.toUpperCase()
}
export const ThemeIconsConstant: ThemeConstantType = {
  dark: 'dark_mode',
  light: 'light_mode'
}

export const ThemeSummaryConstant: ThemeSummaryModel[] = Object.values(ThemeType).map(t => ({
  value: t,
  i18n: ThemeI18NConstant[t],
  icon: ThemeIconsConstant[t]
}))
