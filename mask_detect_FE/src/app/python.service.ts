import { IJson } from './ijson';
import { Observable } from 'rxjs';
import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';

@Injectable({
  providedIn: 'root'
})
export class PythonService {

  // url = 'http://127.0.0.1:5000/';
  url = 'http://innovator.southindia.cloudapp.azure.com:5352/';

  constructor(private httpClient: HttpClient) { }

  postData(img: IJson): Observable<any> {
    // console.log(img);
    return this.httpClient.post<any>(this.url + 'camera', img);
  }

  home(): Observable<any> {
    return this.httpClient.get<any>(this.url);
  }

}
