import {
  ApplicationConfig,
  importProvidersFrom,
  inject,
  provideAppInitializer,
  provideZoneChangeDetection
} from '@angular/core';
import { provideRouter } from '@angular/router';

import { routes } from './app.routes';
import { provideStore } from '@ngrx/store';
import { provideEffects } from '@ngrx/effects';
import { provideRouterStore } from '@ngrx/router-store';

import { HttpClient, provideHttpClient, withInterceptorsFromDi } from '@angular/common/http';
import { TranslateLoader, TranslateModule } from '@ngx-translate/core';
import { TranslateHttpLoader } from '@ngx-translate/http-loader';

import { provideStoreDevtools, StoreDevtoolsModule } from '@ngrx/store-devtools';
import { provideAnimationsAsync } from '@angular/platform-browser/animations/async';
import { MatButtonModule } from '@angular/material/button';
import { MatCardModule } from '@angular/material/card';
import { MatInputModule } from '@angular/material/input';
import { MatFormFieldModule } from '@angular/material/form-field';
import { MatTooltipModule } from '@angular/material/tooltip';
import { BrowserAnimationsModule } from '@angular/platform-browser/animations';
import { settingsReducer } from '@core/store/settings/settings.reducer';
import * as SettingsEffects from '@core/store/settings/settings.effects';
import { browserReducer } from '@core/store/browser/browser.reducer';
import { ThemeService } from '@core/services/settings/theme/theme.service';
import { BrowserService } from './core/services/settings/browser/browser.service';

export function createTranslateLoader(http: HttpClient) {
  return new TranslateHttpLoader(http, './i18n/', '.json');
}

export const appConfig: ApplicationConfig = {
  providers: [
    provideZoneChangeDetection({ eventCoalescing: true }),
    provideStore({
      browser: browserReducer,
      settings: settingsReducer,
    }),
    provideEffects(SettingsEffects),
    provideRouterStore(),
    provideRouter(routes),
    provideStoreDevtools({ logOnly: true }),
    importProvidersFrom([
      BrowserAnimationsModule,
      MatButtonModule,
      MatCardModule,
      MatInputModule,
      MatFormFieldModule,
      MatTooltipModule,
      StoreDevtoolsModule,
      TranslateModule.forRoot({
        loader: {
          provide: TranslateLoader,
          useFactory: createTranslateLoader,
          deps: [HttpClient]
        },
        defaultLanguage: 'en',
      }),
    ]),
    provideAppInitializer(() => {
      inject(BrowserService);
      inject(ThemeService);
    }),
    provideHttpClient(withInterceptorsFromDi()),
    provideAnimationsAsync()
  ]
};
