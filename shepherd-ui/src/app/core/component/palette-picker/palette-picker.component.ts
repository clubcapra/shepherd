import {Component, inject, OnDestroy} from '@angular/core';
import {PalettesSummaryConstant} from '@core/constants/settings/palettes.constant';
import {Store} from '@ngrx/store';
import {MaterialImports} from '../../modules/material-imports.module';
import actions from '@core/store/settings/settings.action';
import {TranslatePipe} from '@ngx-translate/core';
import {PaletteType} from '@core/models/settings/palettes.model';
import {Observable, Subject, takeUntil} from 'rxjs';
import {selectEffectivePalette} from '@core/store/settings/settings.selector';

@Component({
  selector: 'app-palette-picker',
  imports: [MaterialImports, TranslatePipe],
  templateUrl: './palette-picker.component.html',
  styleUrl: './palette-picker.component.scss'
})
export class PalettePickerComponent implements OnDestroy {
  private readonly unsubscribe = new Subject<void>();
  private readonly store = inject(Store);
  protected readonly options = PalettesSummaryConstant;
  protected selected$: Observable<PaletteType> = this.store.select(selectEffectivePalette)
    .pipe(takeUntil(this.unsubscribe));

  onSelect(palette: PaletteType): void {
    this.store.dispatch(actions.changePalette({palette}));
  }

  ngOnDestroy(): void {
    this.unsubscribe.next();
    this.unsubscribe.complete();
  }

  protected readonly Object = Object;
}
