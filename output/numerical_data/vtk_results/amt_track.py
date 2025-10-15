from paraview.simple import *
def start_cue(self):
	global annotations
	global maxIndex
	textSource = paraview.simple.FindSource('Text1')
	if textSource is None:
		Text()
	annotations = []
	annotations.append('24100806_m_E1_trial1')
	annotations.append('24100912_m_E1_trial1')
	annotations.append('24101018_m_E1_trial1')
	annotations.append('24101200_m_E1_trial1')
	annotations.append('24101306_m_E1_trial1')
	annotations.append('24101412_m_E1_trial1')
	maxIndex=len(annotations)
def tick(self):
	global annotations
	global maxIndex
	index = int( self.GetClockTime() )
	if index >= maxIndex :
		index = maxIndex - 1
	textSource = paraview.simple.FindSource('Text1')
	textSource.Text = annotations[index]
def end_cue(self): pass
