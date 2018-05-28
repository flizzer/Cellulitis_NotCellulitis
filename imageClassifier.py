import turicreate as turicreate

images = turicreate.image_analysis.load_images('../images', with_path=True)
images['label'] = images['path'].apply(
    lambda path: 'cellulitis' if '/cellulitis/' in path else 'not_cellulitis'
)

images.groupby('label', [turicreate.aggregate.COUNT])

train_images, test_images = images.random_split(0.8)

model = turicreate.image_classifier.create(
    train_images,
    target='label',
    model='squeezenet_v1.1',
    max_iterations=50 
)

predictions = model.predict(test_images)
metrics = model.evaluate(test_images)
print("Accuracy: {}".format(metrics['accuracy']))
model.save('cellulitis_notcellulitis.model')
model.export_coreml('cellulitis_notcellulitis.mlmodel')