import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface RagMessage {
  id: number;
  content: string;
  timestamp: Date;
}

@Injectable({
  providedIn: 'root'
})
export class RagMessageService {
  private apiUrl = 'https://example.com/api/rag-messages';

  constructor(private http: HttpClient) {}

  getMessages(): Observable<RagMessage[]> {
    return this.http.get<RagMessage[]>(this.apiUrl);
  }

  sendMessage(message: Partial<RagMessage>): Observable<RagMessage> {
    return this.http.post<RagMessage>(this.apiUrl, message);
  }
}
