import {createActionGroup, props} from '@ngrx/store';
import {BrowserState} from '@core/store/browser/browser.model';

const actions = createActionGroup({
  source: 'Browser',
  events: {
    'Update': props<{ values: BrowserState }>(),
    'Change Window Size': props<{ height?: number, width?: number }>(),
  }
});

export default actions;
