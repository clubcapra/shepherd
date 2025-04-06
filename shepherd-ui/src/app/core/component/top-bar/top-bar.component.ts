import { Component, Output, EventEmitter, Input, inject } from '@angular/core';
import { TranslateModule } from '@ngx-translate/core';
import { MaterialImports } from '../../modules/material-imports.module';
import { PalettePickerComponent } from '../palette-picker/palette-picker.component';
import { LanguagePickerComponent } from '../language-picker/language-picker.component';
import { ThemeSelectionComponent } from '@core/component/theme-selection/theme-selection.component';
import { RouterLink } from '@angular/router';
import { Observable } from 'rxjs';
import { Store } from '@ngrx/store';
import { selectBrowserIsMobile } from '../../store/browser/browser.selector';

@Component({
  selector: 'app-top-bar',
  standalone: true,
  imports: [
    TranslateModule,
    MaterialImports,
    PalettePickerComponent,
    LanguagePickerComponent,
    ThemeSelectionComponent,
    RouterLink
  ],
  templateUrl: './top-bar.component.html',
  styleUrls: ['./top-bar.component.scss']
})
export class TopBarComponent {
  @Output() sidenavToggle = new EventEmitter<void>();
  private readonly store = inject(Store);
  isMobile$: Observable<boolean> | undefined;

  ngOnInit(): void {
    this.isMobile$ = this.store.select(selectBrowserIsMobile)
  }

  onToggleSidenav(): void {
    this.sidenavToggle.emit();
  }
}
