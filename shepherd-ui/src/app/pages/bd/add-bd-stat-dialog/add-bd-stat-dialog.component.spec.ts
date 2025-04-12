import { ComponentFixture, TestBed } from '@angular/core/testing';

import { AddBdStatDialogComponent } from './add-bd-stat-dialog.component';

describe('AddBdStatDialogComponent', () => {
  let component: AddBdStatDialogComponent;
  let fixture: ComponentFixture<AddBdStatDialogComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [AddBdStatDialogComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(AddBdStatDialogComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
