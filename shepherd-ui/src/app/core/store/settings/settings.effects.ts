import {inject} from '@angular/core';
import {Actions, createEffect, ofType} from '@ngrx/effects';
import {EMPTY} from 'rxjs';
import {switchMap} from 'rxjs/operators';
import actions from '@core/store/settings/settings.action';
import {ThemeService} from '@core/services/settings/theme/theme.service';

export const changePalette$ = createEffect(
  (
    actions$ = inject(Actions),
    service = inject(ThemeService)
  ) =>
    actions$.pipe(
      ofType(actions.changePalette),
      switchMap(({palette}) => {
        service.palette = palette;
        return EMPTY;
      })
    ),
  {
    dispatch: false,
    functional: true
  }
);

export const changeTheme$ = createEffect(
  (
    actions$ = inject(Actions),
    service= inject(ThemeService)
  ) =>
    actions$.pipe(
      ofType(actions.changeTheme),
      switchMap(({theme}) => {
        service.theme = theme;
        return EMPTY;
      })
    ),
  {
    dispatch: false,
    functional: true
  }
);
