import {createReducer, on} from '@ngrx/store';
import actions from './browser.actions';
import {initialState} from './browser.state';

export const browserReducer = createReducer(
  initialState,
  on(
    actions.update,
    (state, {values}) => ({...state, ...values})
  ),
);
