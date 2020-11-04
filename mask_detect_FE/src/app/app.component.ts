import { IJson } from './ijson';
import { PythonService } from './python.service';
import { Component } from '@angular/core';

@Component({
  selector: 'app-root',
  templateUrl: './app.component.html',
  styleUrls: ['./app.component.scss']
})
export class AppComponent {

  data: IJson;

  title = 'mask-detection';

}
