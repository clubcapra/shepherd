import { inject, Injectable, Inject, PLATFORM_ID } from '@angular/core';
import { BehaviorSubject, fromEvent } from 'rxjs';
import { debounceTime, map } from 'rxjs/operators';
import { BrowserState, DeviceType } from '@core/store/browser/browser.model';
import { Store } from '@ngrx/store';
import actions from '@core/store/browser/browser.actions';

@Injectable({
  providedIn: 'root'
})
export class BrowserService {
  private readonly browserStateSubject: BehaviorSubject<BrowserState>;
  private readonly store: Store = inject(Store);

  constructor(@Inject(PLATFORM_ID) private readonly platformId: object) {
    const initialState = this.deviceInfo;
    this.browserStateSubject = new BehaviorSubject<BrowserState>(initialState);
    if (typeof window !== 'undefined') {
      this.init();
    }
  }

  init() {
    this.store.dispatch(actions.update({ values: this.browserStateSubject.value }));
    fromEvent(window, 'resize')
      .pipe(
        debounceTime(200),
        map(() => this.deviceInfo)
      )
      .subscribe(state => {
        this.browserStateSubject.next(state);
        this.store.dispatch(actions.update({ values: state }));
      });
  }

  private get deviceInfo(): BrowserState {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      return {
        deviceType: DeviceType.Desktop,
        deviceSizeType: DeviceType.Desktop,
        height: 0,
        width: 0,
        userAgent: '',
        isMobile: false,
        touchCapable: false
      };
    }
    return {
      deviceType: this.deviceType,
      deviceSizeType: this.deviceSizeType,
      height: window.innerHeight,
      width: window.innerWidth,
      userAgent: navigator.userAgent,
      isMobile: /Android|webOS|iPhone|iPad|iPod|BlackBerry|IEMobile|Opera Mini/i.test(navigator.userAgent),
      touchCapable: 'ontouchstart' in window || navigator.maxTouchPoints > 0
    };
  }

  private get deviceType(): DeviceType {
    if (typeof window === 'undefined' || typeof navigator === 'undefined') {
      return DeviceType.Desktop;
    }
    const ua = navigator.userAgent.toLowerCase();
    const screenWidth = window.innerWidth;
    const tabletPatterns = [/ipad/i, /tablet/i, /kindle/i, /silk/i, /playbook/i];
    const mobilePatterns = [/android/i, /webos/i, /iphone/i, /ipod/i, /blackberry/i, /windows phone/i];
    const isTablet = tabletPatterns.some(pattern => pattern.test(ua)) ||
      (screenWidth >= 600 && screenWidth < 1024 && ('maxTouchPoints' in navigator && navigator.maxTouchPoints > 0));
    if (isTablet) {
      return DeviceType.Tablet;
    }
    const isMobile = mobilePatterns.some(pattern => pattern.test(ua)) ||
      (screenWidth < 600 && ('maxTouchPoints' in navigator && navigator.maxTouchPoints > 0));
    if (isMobile) {
      return DeviceType.Mobile;
    }
    return DeviceType.Desktop;
  }

  private get deviceSizeType(): DeviceType {
    if (typeof window === 'undefined') {
      return DeviceType.Desktop;
    }
    const screenWidth = window.innerWidth;
    if (screenWidth < 576) {
      return DeviceType.Mobile;
    }
    if (screenWidth < 992) {
      return DeviceType.Tablet;
    }
    return DeviceType.Desktop;
  }
}
