import { Component, inject, OnDestroy, OnInit } from '@angular/core';
import { TranslateModule, TranslateService } from '@ngx-translate/core';
import { Observable, Subject, takeUntil } from 'rxjs';
import { Store } from '@ngrx/store';
import { LocalStorageService } from '@core/services/local-storage/local-storage.service';
import { selectSettingsLanguage } from '@core/store/settings/settings.selector';
import { MaterialImports } from '@core/modules/material-imports.module';
import actionSettings from '@core/store/settings/settings.action';
import { LanguageType } from '@core/models/settings/language.model';
import { TopBarComponent } from './core/component/top-bar/top-bar.component';
import { BottomNavComponent } from './core/component/bottom-nav/bottom-nav.component';
import { RouterModule } from '@angular/router';

@Component({
  selector: 'app-root',
  standalone: true,
  imports: [TranslateModule, MaterialImports, TopBarComponent, BottomNavComponent, RouterModule],
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent implements OnInit, OnDestroy {
  unsubscribe = new Subject<void>();
  language = 'en';
  private readonly localStorageService = inject(LocalStorageService);
  private readonly translate: TranslateService = inject(TranslateService);
  private readonly store = inject(Store);
  isMobile$: Observable<boolean> | undefined;

  ngOnInit(): void {
    const storedLanguage =
      this.localStorageService.getItem<string>('language') ?? 'en';
    if (storedLanguage?.length > 0) {
      this.store.dispatch(
        actionSettings.changeLanguage({
          language: storedLanguage as LanguageType
        })
      );
    } else {
      this.store.dispatch(
        actionSettings.changeLanguage({ language: 'en' as LanguageType })
      );
    }

    this.store
      .select(selectSettingsLanguage)
      .pipe(takeUntil(this.unsubscribe))
      .subscribe((language) => {
        if (!language) return;
        this.language = language;
        this.translate.use(language);
        this.localStorageService.setItem('language', language);
      });
  }

  ngOnDestroy(): void {
    this.unsubscribe.next();
    this.unsubscribe.complete();
  }
}
