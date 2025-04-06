import {createReducer, on} from '@ngrx/store';
import actions from './settings.action';
import {initialState} from './settings.state';

export const settingsReducer = createReducer(
  initialState,
  on(
    actions.changePalette,
    actions.changeTheme,
    actions.changeLanguage,
    actions.changeStickyHeader,
    (state, props) => ({...state, ...props})
  ),
);
