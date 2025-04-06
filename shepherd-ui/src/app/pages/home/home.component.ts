import { MaterialImports } from '@/app/core/modules/material-imports.module';
import {
  RagMessage,
  RagMessageService
} from '@/app/core/services/rag/rag-service';
import { Component } from '@angular/core';
import { FormsModule } from '@angular/forms';
import { RouterModule } from '@angular/router';
import { TranslateModule } from '@ngx-translate/core';

@Component({
  selector: 'app-home',
  imports: [TranslateModule, MaterialImports, RouterModule, FormsModule],
  templateUrl: './home.component.html',
  styleUrl: './home.component.scss'
})
export class HomeComponent {
  messages: RagMessage[] = [];
  newMessage = '';

  constructor(private ragMessageService: RagMessageService) {}

  ngOnInit() {
    this.loadMessages();

    this.messages = [
      {
        timestamp: new Date('2023-04-15T10:00:00'),
        content: 'Détection: Voiture identifiée.',
        id: 0
      },
      {
        timestamp: new Date('2023-04-15T10:05:00'),
        content: 'Détection: Piéton identifié.',
        id: 0
      },
      {
        timestamp: new Date('2023-04-15T10:10:00'),
        content: 'Détection: Cycliste identifié.',
        id: 0
      },
      {
        timestamp: new Date('2023-04-15T10:15:00'),
        content: 'Détection: Feu de signalisation activé.',
        id: 0
      },
      {
        timestamp: new Date('2023-04-15T10:20:00'),
        content: 'Détection: Obstacle sur la route.',
        id: 0
      }
    ];
  }

  loadMessages() {
    this.ragMessageService.getMessages().subscribe((data) => {
      this.messages = data;
    });
  }

  sendMessage() {
    if (this.newMessage.trim()) {
      this.ragMessageService
        .sendMessage({ content: this.newMessage })
        .subscribe((message) => {
          this.messages.push(message);
          this.newMessage = '';
        });
    }
  }
}
