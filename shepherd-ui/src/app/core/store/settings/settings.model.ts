import {ThemeType} from '@core/models/settings/theme.model';
import {PaletteType} from '@core/models/settings/palettes.model';
import {LanguageType} from '@core/models/settings/language.model';

export interface SettingsState {
  language: LanguageType;
  palette: PaletteType;
  theme: ThemeType;
  stickyHeader: boolean;
}

export interface AppState {
  settings: SettingsState;
}
