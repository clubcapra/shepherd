export enum PaletteType {
  DEFAULT= 'default',
  NORD= 'nord'
}

export type PaletteConstantType<T = string> = Record<PaletteType, T>;
export interface PaletteColorModel {
  primary: string;
  secondary: string;
}

export interface PaletteSummary {
  value: PaletteType,
  i18n: string,
  colors: PaletteColorModel
}
