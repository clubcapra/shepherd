/* eslint-disable @typescript-eslint/no-explicit-any */
import { MaterialImports } from '@/app/core/modules/material-imports.module';
import { Component, ViewChild } from '@angular/core';
import { MatDialog } from '@angular/material/dialog';
import { AddBdStatDialogComponent } from './add-bd-stat-dialog/add-bd-stat-dialog.component';
import { MatTableDataSource } from '@angular/material/table';
import { MatSort } from '@angular/material/sort';

@Component({
  selector: 'app-bd',
  imports: [MaterialImports],
  templateUrl: './bd.component.html',
  styleUrl: './bd.component.scss'
})
export class BdComponent {
  displayedColumns = ['name', 'status', 'date', 'precision'];
  dataSource!: MatTableDataSource<any>;

  dbStats = [
    { name: 'Nombre d\'objets uniques', status: 'Appris', date: '24 Juin 2023', precision: 80 },
    { name: 'Chata', status: 'Non appris', date: '20 Déc. 2022', precision: 50 },
    { name: 'Erreur', status: 'Erreur', date: '12 Mai 2023', precision: 30 },
    { name: 'Chat', status: 'Appris', date: '13 Janv. 2023', precision: 90 }
  ];

  @ViewChild(MatSort) sort!: MatSort;

  constructor(private dialog: MatDialog) {}

  ngOnInit(): void {
    this.dataSource = new MatTableDataSource(this.dbStats);
    this.dataSource.filterPredicate = (data, filter: string) => {
      return data.name.trim().toLowerCase().includes(filter);
    };
  }

  ngAfterViewInit(): void {
    this.dataSource.sort = this.sort;
  }

  openAddStatDialog(): void {
    const dialogRef = this.dialog.open(AddBdStatDialogComponent, {
      width: '400px'
    });

    dialogRef.afterClosed().subscribe(result => {
      if (result) {
        this.dbStats = [...this.dbStats, result];
        this.dataSource.data = this.dbStats;
      }
    });
  }

  applyFilter(event: Event): void {
    const filterValue = (event.target as HTMLInputElement).value;
    this.dataSource.filter = filterValue.trim().toLowerCase();
  }
}
