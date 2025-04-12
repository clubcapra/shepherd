import { Component } from '@angular/core';
import { MaterialImports } from '../../modules/material-imports.module';
import { RouterLink, RouterLinkActive } from '@angular/router';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-bottom-nav',
  imports: [MaterialImports, RouterLink, RouterLinkActive, TranslateModule],
  templateUrl: './bottom-nav.component.html',
  styleUrl: './bottom-nav.component.scss'
})
export class BottomNavComponent {

}
