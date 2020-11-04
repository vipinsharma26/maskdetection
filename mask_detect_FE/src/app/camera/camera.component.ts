import { PythonService } from './../python.service';
import { Component, OnInit } from '@angular/core';
import { Subject, Observable } from 'rxjs';
import { WebcamImage, WebcamInitError } from 'ngx-webcam';
import { IJson } from '../ijson';
import { ToastrService } from 'ngx-toastr';
import { NgxSpinnerService } from 'ngx-spinner';
import { toBase64String } from '@angular/compiler/src/output/source_map';
import { DomSanitizer } from '@angular/platform-browser';

@Component({
  selector: 'app-camera',
  templateUrl: './camera.component.html',
  styleUrls: ['./camera.component.scss']
})
export class CameraComponent implements OnInit {

  stopCamera = true;
  reScan = false;
  scaning = false;
  imageURL;

  public showWebcam = true;
  public allowCameraSwitch = true;
  public multipleWebcamsAvailable = false;
  public deviceId: string;

  public errors: WebcamInitError[] = [];

  // latest snapshot
  public webcamImage: WebcamImage = null;

  // webcam snapshot trigger
  private trigger: Subject<void> = new Subject<void>();

  public ngOnInit(): void {
    this.spinnerFunction();
  }

  public triggerSnapshot(): void {
    this.spinner.show();
    this.stopCamera = false;
    this.scaning = true;
    this.trigger.next();
  }

  public handleInitError(error: WebcamInitError): void {
    this.errors.push(error);
  }

  public handleImage(image: WebcamImage): void {
    if (!image) {
      return;
    }

    const newData: IJson = { image } as unknown as IJson;


    this.showWebcam = false;

    this.pythonService.postData(newData).subscribe((data) => {

      this.imageURL = this.sanitizer.bypassSecurityTrustResourceUrl(data.output);

      this.spinner.hide();

      if (data.result === 'Mask Detected') {
        this.toastr.success('Mask Detected');
      } else if (data.result === 'Mask Not Detected') {
        this.toastr.warning('Mask Not Detected');
      }
      this.scan();
    }, (failure) => {
      this.scan();
      this.reScanFun();
      this.toastr.error('Adjust your face in the camera frame', 'Face Not Visible');
    });
  }

  scan(): void {
    this.scaning = false;
    this.reScan = true;
  }

  reScanFun(): void {
    this.stopCamera = true;
    this.reScan = false;
    this.showWebcam = true;
    this.spinnerFunction();
  }

  spinnerFunction():void {
    this.spinner.show();

    setTimeout(() => {
      this.spinner.hide();
    }, 2000);
  }

  public get triggerObservable(): Observable<void> {
    return this.trigger.asObservable();
  }

  constructor(
    private pythonService: PythonService,
    private toastr: ToastrService,
    private spinner: NgxSpinnerService,
    private sanitizer: DomSanitizer
  ) { }
}
