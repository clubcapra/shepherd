/* eslint-disable @typescript-eslint/no-explicit-any */
import { Injectable } from '@angular/core';

const APP_PREFIX = 'dq-';

@Injectable({
  providedIn: 'root'
})
export class LocalStorageService {

  static loadInitialState(): any {
    return Object.keys(localStorage).reduce((state: any, storageKey: string) => {
      if (storageKey.startsWith(APP_PREFIX)) {
        const stateKeys = storageKey
          .replace(APP_PREFIX, '')
          .toLowerCase()
          .split('.')
          .map(key =>
            key
              .split('-')
              .map((token, index) =>
                index === 0
                  ? token
                  : token.charAt(0).toUpperCase() + token.slice(1)
              )
              .join('')
          );
        let currentStateRef = state;
        stateKeys.forEach((key, index) => {
          if (index === stateKeys.length - 1) {
            currentStateRef[key] = JSON.parse(localStorage.getItem(storageKey) ?? 'null');
            return;
          }
          currentStateRef[key] = currentStateRef[key] || {};
          currentStateRef = currentStateRef[key];
        });
      }
      return state;
    }, {});
  }


  getItem<T>(key: string): T | null {
    const item = localStorage.getItem(`${APP_PREFIX}${key}`);
    return item ? JSON.parse(item) : null;
  }

  setItem<T>(key: string, value: T): void {
    localStorage.setItem(`${APP_PREFIX}${key}`, JSON.stringify(value));
  }

  removeItem(key: string): void {
    localStorage.removeItem(`${APP_PREFIX}${key}`);
  }

  testLocalStorage(): void {
    const testValue = 'testValue';
    const testKey = 'testKey';

    this.setItem(testKey, testValue);
    const retrievedValue = this.getItem<string>(testKey);
    this.removeItem(testKey);

    if (retrievedValue !== testValue) {
      throw new Error('localStorage did not return expected value');
    }
  }

  totalSize(): number {
    return Object.keys(localStorage).reduce((total, key) => {
      const item = localStorage.getItem(key);
      return item ? total + item.length : total;
    }, 0);
  }

  availableSpace(): number {
    const MAX_STORAGE = 5_242_880; // 5 MB
    return MAX_STORAGE - this.totalSize();
  }
}
