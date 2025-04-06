import { BrowserState, DeviceType } from '@core/store/browser/browser.model';

export const BROWSER_NAME = 'browser';

export const initialState: BrowserState = {
  height: 0,
  width: 0,
  userAgent: '',
  deviceType: DeviceType.Desktop,
  deviceSizeType: DeviceType.Desktop,
  touchCapable: false,
  isMobile: false,
};
