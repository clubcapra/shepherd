import {Component, OnDestroy, inject} from '@angular/core';
import {Store} from '@ngrx/store';
import {Observable, Subject, takeUntil} from 'rxjs';
import {selectSettingsLanguage} from '@core/store/settings/settings.selector';
import {MaterialImports} from '../../modules/material-imports.module';
import actions from '@core/store/settings/settings.action';
import {
  LanguagesSummaryConstant
} from '@core/constants/settings/languages.constant';
import {TranslatePipe} from '@ngx-translate/core';
import {LanguageType} from '@core/models/settings/language.model';


@Component({
  selector: 'app-language-picker',
  standalone: true,
  imports: [MaterialImports, TranslatePipe],
  templateUrl: './language-picker.component.html',
  styleUrls: ['./language-picker.component.scss']
})

export class LanguagePickerComponent implements  OnDestroy {
  protected options = LanguagesSummaryConstant
  private readonly unsubscribe = new Subject<void>();
  private readonly store = inject(Store);
  protected selected$: Observable<LanguageType>= this.store.select(selectSettingsLanguage)
    .pipe(takeUntil(this.unsubscribe))



  onSelect(language: LanguageType): void {
    this.store.dispatch(actions.changeLanguage({language}));
  }

  ngOnDestroy(): void {
    this.unsubscribe.next();
    this.unsubscribe.complete();
  }
}
