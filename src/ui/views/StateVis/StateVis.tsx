import React from 'react';
import Value from '../../components/Value/Value';

/* it's not clear what this is ATM. */
interface StateVisProps {
    data: any;
}

export default function StateVis({ data }: StateVisProps) {
    return (
        <div className="spector2-vis">
            <Value data={data} />
        </div>
    );
}
