import { createSelector } from '@ngrx/store';
import { SettingsState } from './settings.model';

export const selectSettingsState = (state: AppState) => state.settings;

export interface AppState {
    settings: SettingsState;
}

export const selectSettings = createSelector(
  selectSettingsState,
  (state: SettingsState) => state
);

export const selectSettingsLanguage = createSelector(
  selectSettings,
  (state: SettingsState) => state.language
);

export const selectTheme = createSelector(
  selectSettings,
  settings => settings.theme
);

export const selectPalette = createSelector(
  selectSettings,
  settings => settings.palette.toLowerCase()
);

export const selectSettingsStickyHeader = createSelector(
  selectSettings,
  (state: SettingsState) => state.stickyHeader
);

export const selectEffectivePalette = createSelector(
  selectSettingsState,
  (state: SettingsState) => state.palette
);
