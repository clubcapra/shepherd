import { MaterialImports } from '@/app/core/modules/material-imports.module';
import { Component } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { YoloAddDialogComponent } from './yolo-add-dialog/yolo-add-dialog.component';

@Component({
  selector: 'app-settings',
  imports: [MaterialImports],
  templateUrl: './settings.component.html',
  styleUrl: './settings.component.scss'
})
export class SettingsComponent {
  hyperParamsColumns = ['name', 'value', 'description'];
  yoloColumns = ['className', 'status', 'date', 'precision'];

  hyperparams = [
    {
      name: 'Threshold similarité géométrique',
      value: 7,
      description: 'Description de ce paramètre'
    },
    {
      name: 'Table',
      value: 'Non appris',
      description: 'Description paramètre 2'
    },
    { name: 'Cactus', value: 'Erreur', description: 'Description paramètre 3' },
    { name: 'Chat', value: 'Appris', description: 'Description paramètre 4' }
  ];

  yoloClasses = [
    {
      className: 'Chaise',
      status: 'Appris',
      date: '24 Juin 2023',
      precision: 80
    },
    {
      className: 'Table',
      status: 'Non appris',
      date: '20 Déc. 2022',
      precision: 50
    },
    {
      className: 'Cactus',
      status: 'Erreur',
      date: '12 Mai 2023',
      precision: 30
    },
    {
      className: 'Chat',
      status: 'Appris',
      date: '13 Janv. 2023',
      precision: 90
    }
  ];

  constructor(private dialog: MatDialog) {}

  updateParamValue(element: any, newValue: any) {
    //IMPLEMENT BACKEND STUFF HERE
    console.log('Updating param value:', element.name, 'to', newValue);
  }

  openAddYoloDialog(): void {
    const dialogRef = this.dialog.open(YoloAddDialogComponent, {
      width: '300px'
    });

    dialogRef.afterClosed().subscribe((result) => {
      if (result) {
        const newClass = {
          className: result.className,
          status: 'Non appris',
          date: new Date().toLocaleDateString(),
          precision: 0
        };
        this.yoloClasses = [...this.yoloClasses, newClass];
      }
    });
  }
}
