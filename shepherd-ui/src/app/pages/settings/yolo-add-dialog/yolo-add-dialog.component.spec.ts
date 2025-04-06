import { ComponentFixture, TestBed } from '@angular/core/testing';

import { YoloAddDialogComponent } from './yolo-add-dialog.component';

describe('YoloAddDialogComponent', () => {
  let component: YoloAddDialogComponent;
  let fixture: ComponentFixture<YoloAddDialogComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      imports: [YoloAddDialogComponent]
    })
    .compileComponents();

    fixture = TestBed.createComponent(YoloAddDialogComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
