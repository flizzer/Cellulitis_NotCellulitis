 //
//  ViewController.swift
//  Cellulitis_NotCellulitis
//
//  Created by Brian Davis on 5/28/18.
//  Copyright © 2018 Brian Davis. All rights reserved.
//

import UIKit
import CoreML
import Vision

class ViewController: UIViewController
    , UIImagePickerControllerDelegate
    , UINavigationControllerDelegate {

    @IBOutlet weak var imageView: UIImageView!
    
    let imagePicker = UIImagePickerController()
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        imagePicker.delegate = self
        imagePicker.sourceType = .photoLibrary
        imagePicker.allowsEditing = false
    }

    @IBAction func cameraTapped(_ sender: UIBarButtonItem) {
        present(imagePicker, animated: true,  completion: nil)
    }
    
    func imagePickerController(_ picker: UIImagePickerController
        , didFinishPickingMediaWithInfo info: [String : Any]) {
        if let imagePicked = info[UIImagePickerControllerOriginalImage]
            as? UIImage {
                imageView.image = imagePicked
            
            guard let ciImage = CIImage(image: imagePicked) else {
                fatalError("Could not convert UIImage to CIImage!")
            }
            
            detect(ciImage: ciImage)
        }
        imagePicker.dismiss(animated: true, completion: nil)
    }
    
    func detect(ciImage: CIImage)
    {
        guard let model = try? VNCoreMLModel(
            for: cellulitis_notcellulitis().model) else {
            fatalError("Loading of Cellulitis_NotCellulitis Model failed!")
        }
        
        let request = VNCoreMLRequest(model: model) {
            (request, error) in
                guard let results = request.results
                    as? [VNClassificationObservation]
                else
                {
                    fatalError("Could not get request results!")
                }
            
            if let classification = results.first {
                if classification.identifier == "cellulitis" {
                    self.navigationItem.title =
                        "Cellulitis with a confidence of \(classification.confidence)!"
                }
                else
                {
                    self.navigationItem.title =
                        "Not Cellulitis with a confidence of \(classification.confidence)!"
                }
            }
        }
        
        let handler = VNImageRequestHandler(ciImage: ciImage)
        
        do {
            try handler.perform([request])
        }
        catch {
            print(error)
        }
        
    }
    
    
}

