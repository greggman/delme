import React from 'react';
import Debugger from '../Debugger/Debugger';
import MiniUI from '../MiniUI/MiniUI';
import { createUIState, UIProps, UIState, UIStateContext } from '../../contexts/UIStateContext';

import BufferVis from '../../views/BufferVis/BufferVis';
import FramesVis from '../../views/FramesVis/FramesVis';
import ResultVis from '../../views/ResultVis/ResultVis';
import StepsVis from '../../views/StepsVis/StepsVis';

import './App.css';

class App extends React.Component<UIProps, UIState> {
    constructor(props: UIProps) {
        super(props);
        const { uiStateHelper } = props;
        this.state = createUIState({
            paneIdToViewType: {
                pane0: { component: FramesVis, data: ['frame1', 'frame2'] },
                pane1: { component: StepsVis, data: null },
                pane2: { component: ResultVis, data: null },
                pane3: { component: BufferVis, data: null },
            },
        });
        uiStateHelper.setStateFn = (...args) => {
            this.setState(...args);
        };
    }
    render() {
        const { uiStateHelper } = this.props;
        uiStateHelper.updateState(this.state);
        return (
            <UIStateContext.Provider value={{ helper: uiStateHelper }}>
                <div className="spector2">{this.state.fullUI ? <Debugger /> : <MiniUI />}</div>
            </UIStateContext.Provider>
        );
    }
}

export default App;
