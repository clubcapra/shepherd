import { Routes } from '@angular/router';
import { HomeComponent } from './pages/home/home.component';
import { SettingsComponent } from './pages/settings/settings.component';
import { BdComponent } from './pages/bd/bd.component';

export const routes: Routes = [
    { path: 'home', component: HomeComponent },
    { path: 'settings', component: SettingsComponent },
    { path: 'bd', component: BdComponent },
    { path: '', redirectTo: 'home', pathMatch: 'full' },
];
