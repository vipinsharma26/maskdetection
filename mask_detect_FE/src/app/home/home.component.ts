import { Component, OnInit } from '@angular/core';
import { PythonService } from '../python.service';

@Component({
  selector: 'app-home',
  templateUrl: './home.component.html',
  styleUrls: ['./home.component.scss']
})
export class HomeComponent implements OnInit {

  welcome: string;

  constructor(
    private pythonService: PythonService,
  ) { }

  ngOnInit(): void {
    this.pythonService.home().subscribe((data) => {
      this.welcome = data.test;
    });
  }

}
