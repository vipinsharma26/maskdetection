import { Component, OnInit, ElementRef, ViewChild } from '@angular/core';

@Component({
  selector: 'app-feedback',
  templateUrl: './feedback.component.html',
  styleUrls: ['./feedback.component.scss']
})
export class FeedbackComponent implements OnInit {

  saveFile = false;

  constructor() { }

  changeEvent(event): void {
    console.log(event.target.checked);
    if (event.target.checked) {
      this.saveFile = true;
    }
    else {
      this.saveFile = false;
    }
  }

  handleUpload(event): void {
    const file = event.target.files[0];
    const reader = new FileReader();
    reader.readAsDataURL(file);
    reader.onload = () => {
      console.log(reader.result);
    };
  }

  ngOnInit(): void {
  }

}
