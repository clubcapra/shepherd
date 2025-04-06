import { createFeatureSelector, createSelector } from '@ngrx/store';
import { BrowserState } from './browser.model';
import { BROWSER_NAME } from './browser.state';

export const selectBrowserState = createFeatureSelector<BrowserState>(BROWSER_NAME);
export const selectBrowserHeight = createSelector(selectBrowserState, (state: BrowserState) => state.height);
export const selectBrowserWidth = createSelector(selectBrowserState, (state: BrowserState) => state.width);
export const selectBrowserUserAgent = createSelector(selectBrowserState, (state: BrowserState) => state.userAgent);
export const selectBrowserDeviceType = createSelector(selectBrowserState, (state: BrowserState) => state.deviceType);
export const selectBrowserDeviceSizeType = createSelector(selectBrowserState, (state: BrowserState) => state.deviceSizeType);
export const selectBrowserTouchCapable = createSelector(selectBrowserState, (state: BrowserState) => state.touchCapable);
export const selectBrowserIsMobile = createSelector(selectBrowserState, (state: BrowserState) => state.isMobile);