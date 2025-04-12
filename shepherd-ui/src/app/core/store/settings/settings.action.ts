import {createActionGroup, props} from '@ngrx/store';
import {ThemeType} from '@core/models/settings/theme.model';
import {PaletteType} from '@core/models/settings/palettes.model';
import {LanguageType} from '@core/models/settings/language.model';

const actions = createActionGroup({
  source: 'Settings',
  events: {
    'Change Palette': props<{ palette: PaletteType }>(),
    'Change Language': props<{language: LanguageType}>(),
    'Change Theme': props<{theme: ThemeType}>(),
    'Change Sticky Header': props<{ stickyHeader: boolean }>()
  }
});

export default actions;
