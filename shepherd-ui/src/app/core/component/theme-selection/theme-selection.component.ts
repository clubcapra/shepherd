import {Component, computed, inject, OnDestroy, OnInit, signal} from '@angular/core';
import {Subject, takeUntil} from 'rxjs';
import {Store} from '@ngrx/store';
import {MatIconButton} from '@angular/material/button';
import {MatIcon} from '@angular/material/icon';
import actions from '@core/store/settings/settings.action';
import {selectTheme} from '@core/store/settings/settings.selector';
import {MatTooltip} from '@angular/material/tooltip';
import {ThemeIconsConstant, ThemeConstant, ThemeI18NConstant} from '@core/constants/settings/theme.constant';
import {ThemeType} from '@core/models/settings/theme.model';
import {TranslatePipe} from '@ngx-translate/core';

@Component({
  selector: 'app-theme-selection',
  imports: [
    MatIconButton,
    MatIcon,
    MatTooltip,
    TranslatePipe
  ],
  templateUrl: './theme-selection.component.html',
  styleUrl: './theme-selection.component.scss'
})
export class ThemeSelectionComponent implements OnInit, OnDestroy {
  private readonly unsubscribe = new Subject<void>();
  private readonly store = inject(Store);

  protected readonly options = ThemeConstant;
  protected readonly optionsIcons = ThemeIconsConstant;
  protected readonly optionsI18N = ThemeI18NConstant;
  protected theme$ = signal<ThemeType>(this.options[0]);
  protected nextThemeIndex = computed(() =>
    (this.options.findIndex(o => o === this.theme) + 1) % this.options.length
  );

  get nextTheme() {
    return this.options[this.nextThemeIndex()];
  }

  get theme(): ThemeType {
    return this.theme$();
  }

  ngOnInit(): void {
    this.store.select(selectTheme)
      .pipe(takeUntil(this.unsubscribe))
      .subscribe((theme) => {
        this.theme$.set(theme);
      });
  }

  toggleTheme(): void {
    this.store.dispatch(actions.changeTheme({theme: this.nextTheme}));
  }

  ngOnDestroy(): void {
    this.unsubscribe.next();
    this.unsubscribe.complete();
  }
}
