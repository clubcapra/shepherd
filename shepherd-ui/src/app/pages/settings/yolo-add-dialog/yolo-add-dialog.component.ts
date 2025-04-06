import { MaterialImports } from '@/app/core/modules/material-imports.module';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { MatDialogRef } from '@angular/material/dialog';

@Component({
  selector: 'app-yolo-add-dialog',
  imports: [FormsModule, MaterialImports],
  templateUrl: './yolo-add-dialog.component.html',
  styleUrl: './yolo-add-dialog.component.scss'
})
export class YoloAddDialogComponent {
  className = '';

  constructor(public dialogRef: MatDialogRef<YoloAddDialogComponent>) {}

  onCancel(): void {
    this.dialogRef.close();
  }

  onAdd(): void {
    this.dialogRef.close({ className: this.className });
  }
}
