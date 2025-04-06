export enum DeviceType {
  Desktop = 'desktop',
  Tablet = 'tablet',
  Mobile = 'mobile'
}

export interface BrowserState {
  height: number;
  width: number;
  userAgent: string;
  deviceSizeType: DeviceType,
  deviceType: DeviceType,
  touchCapable: boolean;
  isMobile: boolean;

}
