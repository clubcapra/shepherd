import {
  PaletteColorModel,
  PaletteConstantType,
  PaletteSummary,
  PaletteType
} from '@core/models/settings/palettes.model';

export const PalettesConstant = Object.values(PaletteType);

export const PalettesI18NConstant: PaletteConstantType = {
  default: 'palette.default'.toUpperCase(),
  nord: 'palette.nord'.toUpperCase()
};

export const PalettesColorConstant: PaletteConstantType<PaletteColorModel> = {
  default: {
    primary: '#518252',
    secondary: '#d9dbd5'
  },
  nord: {
    primary: '#537f7e',
    secondary: '#d0daf2'
  }
};

export const PalettesSummaryConstant: PaletteSummary[] = Object.values(PaletteType).map(t => ({
  value: t,
  i18n: PalettesI18NConstant[t],
  colors: PalettesColorConstant[t]
}));
