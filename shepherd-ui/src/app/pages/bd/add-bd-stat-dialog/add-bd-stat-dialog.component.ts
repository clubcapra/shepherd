import { MaterialImports } from '@/app/core/modules/material-imports.module';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-add-bd-stat-dialog',
  imports: [MaterialImports, FormsModule],
  templateUrl: './add-bd-stat-dialog.component.html',
  styleUrl: './add-bd-stat-dialog.component.scss'
})
export class AddBdStatDialogComponent {
  name = '';
  status = 'Appris';
  date = new Date().toLocaleDateString('fr-FR');
  precision = 0;

  constructor(private dialogRef: MatDialogRef<AddBdStatDialogComponent>) {}

  onCancel(): void {
    this.dialogRef.close();
  }

  onAdd(): void {
    this.dialogRef.close({
      name: this.name,
      status: this.status,
      date: this.date,
      precision: this.precision
    });
  }
}
