import { SettingsState } from '@core/store/settings/settings.model';
import { ThemeType } from '@core/models/settings/theme.model';
import { LanguageType } from '@core/models/settings/language.model';
import { PaletteType } from '@core/models/settings/palettes.model';

const defaultTheme = (() => {
  if (typeof window !== 'undefined') {
    return window.matchMedia('(prefers-color-scheme: dark)').matches
      ? ThemeType.DARK
      : ThemeType.LIGHT;
  }
  return ThemeType.LIGHT;
})();

export const initialState: SettingsState = {
  language: LanguageType.EN,
  palette: PaletteType.DEFAULT,
  stickyHeader: true,
  theme: defaultTheme
};
