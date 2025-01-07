
# Import necessary libraries
import matplotlib.pyplot as plt


# Define the classifiers, performance metrics, and corresponding values
classifiers = ['Decision Tree', 'SVM', 'K-NN', 'Random Forest', 'MLP', 'Naive Bayes', 'Extreme Learning Machine']
sensitivity_scores = [0.10, 0.73, 0.89, 0.61, 0.10, 0.87, 0.87]  # Example sensitivity scores
specificity_scores = [0.70, 0.79, 0.69, 0.92, 0.92, 0.87, 0.95]  # Example specificity scores
f1_scores = [0.50, 0.70, 0.65, 0.58, 0.50, 0.63, 0.88]  # Example F1 scores
accuracy_scores = [0.72, 0.77, 0.76, 0.72, 0.74, 0.77, 0.92]  # Example accuracy scores

# Create a figure and axis
fig, ax = plt.subplots(figsize=(12, 8))

# Plot the grouped bar diagram for sensitivity, specificity, F1 score, and accuracy
barWidth = 0.15
r1 = range(len(classifiers))
r2 = [x + barWidth for x in r1]
r3 = [x + barWidth for x in r2]
r4 = [x + barWidth for x in r3]

plt.bar(r1, sensitivity_scores, color='skyblue', width=barWidth, edgecolor='grey', label='Sensitivity')
plt.bar(r2, specificity_scores, color='salmon', width=barWidth, edgecolor='grey', label='Specificity')
plt.bar(r3, f1_scores, color='lightgreen', width=barWidth, edgecolor='grey', label='F1 Score')
plt.bar(r4, accuracy_scores, color='orange', width=barWidth, edgecolor='grey', label='Accuracy')

# Add xticks on the middle of the group bars
plt.xlabel('Classifiers', fontweight='bold')
plt.xticks([r + 1.5*barWidth for r in range(len(classifiers))], classifiers)

# Set the title and labels
plt.title('COMPARISON OF CLASSIFIERS WITH EXTREME LEARNING MACHINE')
plt.ylabel('Score')
plt.legend()

# Show the plot
plt.show()
