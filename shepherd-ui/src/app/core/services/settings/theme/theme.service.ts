import { inject, Injectable } from '@angular/core';
import { LocalStorageService } from '@core/services/local-storage/local-storage.service';

import { PalettesConstant } from '@core/constants/settings/palettes.constant';
import { ThemeType } from '@core/models/settings/theme.model';
import { ThemeConstant } from '@core/constants/settings/theme.constant';
import { PaletteType } from '@core/models/settings/palettes.model';

import actions from '@core/store/settings/settings.action';
import { Store } from '@ngrx/store';

import { isPlatformBrowser } from '@angular/common';
import { Inject, PLATFORM_ID } from '@angular/core';

@Injectable({
  providedIn: 'root'
})
export class ThemeService {
  private readonly localStorageService = inject(LocalStorageService);
  private readonly themeOptions: ThemeType[] = ThemeConstant;
  private readonly availablePalette = PalettesConstant;
  private selectedTheme?: ThemeType = undefined;
  private selectedPalette?: PaletteType = undefined;

  constructor(private readonly store: Store, @Inject(PLATFORM_ID) private readonly platformId: object) {
    if (isPlatformBrowser(this.platformId)) {
      this.store.dispatch(actions.changeTheme({ theme: this.theme }));
      this.store.dispatch(actions.changePalette({ palette: this.palette }));
    }
  }

  get palette(): PaletteType {
    const storedPalette = (this.localStorageService.getItem('palette') ?? this.availablePalette[0]) as PaletteType;
    this.palette = storedPalette;
    return storedPalette;
  }

  set palette(palette: PaletteType) {
    if (this.selectedPalette === palette) return;
    this.selectedPalette = palette;
    this.localStorageService.setItem('palette', palette);

    const bodyClasses = document.body.className.split(' ');
    bodyClasses.forEach((className) => {
      if (this.availablePalette.some(p => (p + '-palette') === className))
        document.body.classList.remove(className);
    });
    document.body.classList.add(palette + '-palette');
  }

  get theme(): ThemeType {
    const storedTheme = this.localStorageService.getItem('theme');
    if (storedTheme) {
      this.theme = storedTheme as ThemeType;
      return storedTheme as ThemeType;
    }
    if (typeof window !== 'undefined' && window.matchMedia) {
      const preferred = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
      this.theme = preferred as ThemeType;
      return preferred as ThemeType;
    }
    return 'light' as ThemeType;
  }

  set theme(theme: ThemeType) {
    if (this.selectedTheme === theme) return;
    this.selectedTheme = theme;
    this.localStorageService.setItem('theme', theme);

    const bodyClasses = document.body.className.split(' ');
    bodyClasses.forEach((className) => {
      if (this.themeOptions.some(t => (t + '-theme') === className))
        document.body.classList.remove(className);
    });
    document.body.classList.add(theme + '-theme');
  }
}
